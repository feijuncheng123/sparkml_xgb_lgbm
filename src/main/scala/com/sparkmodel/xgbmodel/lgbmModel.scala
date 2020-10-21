package com.sparkmodel.xgbmodel
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
//import com.microsoft.ml.spark.LightGBMClassifier
import com.microsoft.ml.spark.lightgbm.LightGBMClassifier
import org.apache.spark.ml.param.{DoubleParam, Param}


object lgbmModel {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark: SparkSession =SparkSession.builder().appName("H2OModelBuilding")
        .enableHiveSupport().getOrCreate()
    import spark.implicits._

    val tools=new lgbmModel(spark)
    val data=tools.getModelData

    val modelFeatures=(data.columns :+ "brand_index").filterNot(Array("mbl_no","newflag","brand").contains(_))

    val stringIndexStages=new StringIndexer().setInputCol("brand").setOutputCol("brand_index")
    val vectorAssembleStage=new VectorAssembler()
        .setInputCols(modelFeatures)
        .setOutputCol("features")
    val vectorIndexer=new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(100)
    val pip=new Pipeline().setStages(Array(stringIndexStages,vectorAssembleStage,vectorIndexer))
    val modelData=pip.fit(data).transform(data)

    val Array(trainingAllData,testdt)=modelData.randomSplit(Array(0.8,0.2),888888L)
    val positiveSample=trainingAllData.filter("newflag=0")
    val negativeSample=trainingAllData.filter("newflag=1").sample(false,0.02,seed = 888888L)
    val trainingData=positiveSample.union(negativeSample)
    trainingData.cache()
    tools.logger.info(s"样本总数为：")
    trainingData.groupBy("newflag").count().show()

    tools.logger.info("开始训练模型-------------------------------------------------------------")
    val lgbm=new LightGBMClassifier()
        .setBoostingType("goss")
        .setObjective("binary")
        .setNumIterations(300)
        .setFeaturesCol("indexedFeatures")
        .setLabelCol("newflag")
        .setPredictionCol("prediction")
        .setProbabilityCol("probability")
        .setEarlyStoppingRound(20)
        .setFeatureFraction(0.5)
        .setLambdaL2(1.0)
        .setMaxDepth(9)
        .setLearningRate(0.1)

    val lgbmModel=lgbm.fit(trainingData)
    val imp=lgbmModel.getFeatureImportances("gain")
    modelFeatures.zip(imp).foreach{case (a,b)=>println(s"$a:$b")}

    val predictions = lgbmModel.transform(testdt)
    val check=predictions.select("newflag","prediction")
    check.cache()
    val truePositive=check.filter("is_fins_face_rec=1").filter("prediction=1").count()
    val falsePositive=check.filter("is_fins_face_rec=0").filter("prediction=1").count()
    val trueNegative=check.filter("is_fins_face_rec=0").filter("prediction=0").count()
    val falseNegative=check.filter("is_fins_face_rec=1").filter("prediction=0").count()

    val precision=truePositive.toDouble/(truePositive+falsePositive).toDouble
    val recall=truePositive.toDouble/(truePositive+falseNegative).toDouble
    println(s"precision=$precision,recall=$recall")

    spark.stop()
  }

}

class lgbmModel(@transient val spark:SparkSession) extends java.io.Serializable{
  import spark.implicits._
  @transient val logger: Logger =LogManager.getLogger(this.getClass)

  val dropCols=Array(Array("prov_cd","prov_name","city_cd","city_name","is_natl_roam","sex_cd",
    "basic_name","usr_sts_cd","is_parent","is_child","usr_rsk_value", "last_chg_term_tm",
    "imei","is_new_term", "opendata_req_flag","is_section_no_yd","app6","app10","app11","battery_typ_cd",
    "app18","app22","app30","app34","app36","app37","app38","app40","app42","app44","app47","app53",
    "app61","app62","app63","app66","app68","app73","app6_uniq","app10_uniq","app11_uniq","app18_uniq",
    "app22_uniq","app30_uniq","app34_uniq","app36_uniq","app37_uniq","app38_uniq","app40_uniq",
    "app42_uniq","app44_uniq","app47_uniq","app53_uniq","app61_uniq","app62_uniq","app63_uniq",
    "app66_uniq","app68_uniq","app73_uniq","section_no","type1",
    "age","pron_cd","term_typ_cd","term_ol_tm","os_cd","os_detl","os_typ_cd","net_patn_cd","card_patn_cd","cutn_info_cd",
    "term_style_cd","scr_sz","scr_reso","cpu_core","cpu_info","touch_typ_cd",
    "touch_way_cd","imp_typ_cd","sell_area_cd","term_mkt_rltm_amt","chg_ago_term_mkt_amt","chg_ago_term_mkt_rltm_amt",
    "mobile_body_rom","mobile_rom","cm_sms_use_cnt",
    "exp_mem","battery_vol","sim_type","main_pix","front_pix","wait_time_cd","is_cunt",
    "is__spt_bluetooth","is_spt_nfc","is_spt_fing","chg_ago_term_type","chg_ago_term_brand",
    "chg_ago_term_sys","chg_ago_term_net_pant","last_term_time","term_model","term_fac","is_139_bill_read"
  ),
    Array("term_brand","provinces","citys","flag","city_level")
  )

  val categoryCols=Array("pck_brand_cd","usr_star_cd","is_owe","cm_mm_expe_trait_cd",
    "is_real_name_usr","is_low_qty_usr","is_marry","is_college_student","is_departure","is_vnet_usr",
    "data_iral","is_uthin_ph","is_bscr_ph","is_fm_ph","is_camera_ph","is_thr_anti_ph","is_om_ph_ph",
    "is_stu_ph","is_mc_ph","is_pb_ph","is_sw_ph","is_gm_ph","is_spt_dsds","is_spt_fc","is_spt_cur_dis",
    "is_spt_volte_call","term_use_dur","family_vnet","campus_vnet","is_company_employee","cm_use_flow",
    "cm_call_tot_dur","cm_waln_flow","cm_pck_fixed_fee","cm_arpu","cm_ivr_fee","cm_addvalue_fee",
    "last3_mon_avg_arpu","last3_mon_avg_fee","last3_mon_avg_vas_fee","last6_mon_owe_cnt",
    "last6_mon_roam_cnt","cm_fee_bal_fee","cm_acct_fee","cm_ref_bal_fee","cm_local_ivr_fee",
    "cm_intl_ivr_fee","cm_pck_out_intl_ivr_fee","cm_oth_pay_fee","cm_sms_fee",
    "cm_pck_out_flow_fee_sub", "cm_owe_amt_sub","brand","newflag"
  )

  def getModelData: DataFrame ={
    val tagData=spark.sql("select * from mk1.tmp_vip_model_dt")
        .withColumn("brand",
          when($"term_brand".isNull,"未知品牌")
              .when($"term_brand".isin("OPPO","vivo","华为","苹果","华为荣耀","小米",
                "三星","魅族","金立","中国移动","联想","酷派","诺基亚","中兴","乐视","360","一加","美图","锤子","中兴努比亚",
                "海信","小天才","酷比","朵唯","天语","小辣椒","TCL","摩托罗拉","康佳","贝尔丰","HTC","盛泰","酷聊","乐丰",
                "百立丰","索尼爱立信","华硕","长虹","有方","诺亚信","金国威","邦华","糖果","波导","欧正","兴华宝","海尔","亿通",
                "先科","黑鲨"),$"term_brand").otherwise("小品牌"))
        .withColumn("newflag",when($"flag">0,0)
            .when($"flag"===0,1))

    val modelData=tagData.na.fill(0,tagData.columns.filter(_ != "newflag"))
    val modelCols=modelData.columns.filter(!dropCols.flatten.contains(_))
    val sampleData=modelData.select(modelCols.map(col):_*)

    //    val Array(trainingAllData,testdt)=sampleData.randomSplit(Array(0.8,0.2),888888L)
    //
    //    val positiveSample=trainingAllData.filter("newflag=0")
    //    val negativeSample=trainingAllData.filter("newflag=1").sample(false,0.01,seed = 888888L)

    //    val trainingData=positiveSample.union(negativeSample)
    //    trainingData.printSchema()
    //    logger.info(s"样本总数为：")
    //    trainingData.groupBy("newflag").count().show()

    //    (trainingData,testdt)
    sampleData
  }

}