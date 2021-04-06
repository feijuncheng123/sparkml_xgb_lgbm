package com.sparkmodel.xgbmodel

import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.sql.types.IntegerType
//import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, lit, when}
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
//import ml.dmlc.xgboost4j.scala.{XGBoost,DMatrix}


object xgbmodel {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val spark = SparkSession
        .builder()
        .appName("XGBoost")
        //        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .enableHiveSupport()
        .getOrCreate()

    val tools=new xgbmodel(spark)

    ///read data from disk
    val data=tools.getModelData
    val modelFeatures=(data.columns :+ "brand_index").filterNot(Array("mbl_no","label","brand").contains(_))
    val stringIndexStages=new StringIndexer().setInputCol("brand").setOutputCol("brand_index")
    val vectorAssembleStage=new VectorAssembler()
        .setInputCols(modelFeatures)
        .setOutputCol("features")

    val vectorIndexer=new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(10)
    val pip=new Pipeline().setStages(Array(stringIndexStages,vectorAssembleStage,vectorIndexer))
    val modelData=pip.fit(data).transform(data)

    val Array(trainingAllData,testdt)=modelData.randomSplit(Array(0.8,0.2),888888L)
    val positiveSample=trainingAllData.filter("label=1")
    val negativeSample=trainingAllData.filter("label=0").sample(false,0.3,seed = 888888L)
    val trainingData=positiveSample.union(negativeSample)

    trainingData.cache()
    tools.logger.info(s"样本总数为：")
    trainingData.groupBy("label").count().show()

    tools.logger.info("开始训练模型-------------------------------------------------------------")
    // Split the data into training and test sets (30% held out for testing).
    //    val Array(trainingData, testData) = train.randomSplit(Array(0.7, 0.3), seed = 0)

    // number of iterations
    val numRound = 200
    val numWorkers = 10
    val paramMap = List(
      "booster" -> "gbtree",
      "eta" -> 0.1f,
      "max_depth" -> 10,
      "subsample" -> 1,
      "colsample_bytree" -> 0.7,
      "lambda" -> 1,
      "grow_policy" -> "lossguide",
      "eval_metric" -> "auc",
      "silent" -> 1,
      "objective" -> "binary:logistic").toMap
    println("Starting Xgboost ")

    val xgBoostModelWithDF =new XGBoostClassifier(paramMap)
//        .setLabelCol("label")
        .setLabelCol("newflag")
        .setFeaturesCol("indexedFeatures")
        .setProbabilityCol("probability")
        .setPredictionCol("prediction")
        .setNumRound(300)
        .setNumWorkers(10)
        .setNthread(1)
        .setUseExternalMemory(true)
        .setNumEarlyStoppingRounds(20)
        .setMaximizeEvaluationMetrics(true)
        .setSeed(49L)
        .setTrainTestRatio(0.3)


    val xgbmodel=xgBoostModelWithDF.fit(trainingData)

    xgbmodel.write.overwrite().save("hdfs:///user/fjc_model/xgbModel")

    val predictions = xgbmodel.transform(testdt)


    val featureImportance=xgbmodel.nativeBooster.getFeatureScore()
    featureImportance.foreach{case (a:String,b:Integer)=>println(a+"|"+b)}

    //这种方法比较好，也是获取字段重要性
    xgbmodel.nativeBooster.getScore("", "gain")

    //    DataFrames can be saved as Parquet files, maintaining the schema information
    //    predictions.write.save("preds.parquet")
    tools.logger.info("混淆矩阵-------------------------------------------------------------")
    predictions.groupBy("label").pivot("prediction").count().show()
//    predictions.show(20,false)
    tools.logger.info("建模完毕-------------------------------------------------------------")
    //    prediction on test set for submission file
    //    val submission = xgBoostModelWithDF.setExternalMemory(true).transform(testdt).select("id", "probabilities")
    //    submission.show(10)
    //    submission.write.save("submission.parquet")
    spark.stop()
  }
}


class xgbmodel(@transient  val spark:SparkSession) extends java.io.Serializable{
  import spark.implicits._
  @transient val logger: Logger =LogManager.getLogger(this.getClass)

  val dropCols=Array(Array("prov_cd","prov_name","city_cd","city_name","is_natl_roam",
    "basic_name","usr_sts_cd","is_parent","is_child","usr_rsk_value", "last_chg_term_tm",
    "imei","is_new_term", "opendata_req_flag","is_section_no_yd","battery_typ_cd",
    "section_no","type1","app1","app13","app22","app23","app30","app37","app38","app40","app43",
    "app44","app45","app55","app59","app60","app61","app71","app72","app74","app75","app76",
    "app77","app79","app80","app85","app86","app88","app1_uniq","app13_uniq","app22_uniq",
    "app23_uniq","app30_uniq","app37_uniq","app38_uniq","app40_uniq","app43_uniq","app44_uniq",
    "app45_uniq","app55_uniq","app59_uniq","app60_uniq","app61_uniq","app71_uniq","app72_uniq",
    "app74_uniq","app75_uniq","app76_uniq","app77_uniq","app79_uniq","app80_uniq","app85_uniq",
    "app86_uniq","app88_uniq",
    "age","pron_cd","term_typ_cd","term_ol_tm","os_cd","os_detl","os_typ_cd","net_patn_cd","card_patn_cd","cutn_info_cd",
    "term_style_cd","scr_sz","scr_reso","cpu_core","cpu_info","touch_typ_cd",
    "touch_way_cd","imp_typ_cd","sell_area_cd","term_mkt_rltm_amt","chg_ago_term_mkt_amt","chg_ago_term_mkt_rltm_amt",
    "mobile_body_rom","mobile_rom",
    "exp_mem","battery_vol","sim_type","main_pix","front_pix","wait_time_cd","is_cunt",
    "is__spt_bluetooth","is_spt_nfc","is_spt_fing","chg_ago_term_type","chg_ago_term_brand",
    "chg_ago_term_sys","chg_ago_term_net_pant","last_term_time","term_model","term_fac","is_139_bill_read"
  ),
    Array("term_brand","provinces","citys","city_level","sex_cd")
  )

  val categoryCols=Array("pck_brand_cd","usr_star_cd","is_owe","cm_mm_expe_trait_cd",
    "is_real_name_usr","is_low_qty_usr","is_marry","is_college_student","is_departure","is_vnet_usr",
    "data_iral","is_uthin_ph","is_bscr_ph","is_fm_ph","is_camera_ph","is_thr_anti_ph","is_om_ph_ph",
    "is_stu_ph","is_mc_ph","is_pb_ph","is_sw_ph","is_gm_ph","is_spt_dsds","is_spt_fc","is_spt_cur_dis",
    "is_spt_volte_call","term_use_dur","family_vnet","campus_vnet","is_company_employee",
    "brand"
  )


  def getModelData: DataFrame ={
    val tagData=spark.sql("select * from mk1.tmp_final_vip_model_dt")
        .withColumn("brand",
          when($"term_brand".isNull,"未知品牌")
              .when($"term_brand".isin("OPPO","vivo","华为","苹果","华为荣耀","小米",
                "三星","魅族","金立","中国移动","联想","酷派","诺基亚","中兴","乐视","360","一加","美图","锤子","中兴努比亚",
                "海信","小天才","酷比","朵唯","天语","小辣椒","TCL","摩托罗拉","康佳","贝尔丰","HTC","盛泰","酷聊","乐丰",
                "百立丰","索尼爱立信","华硕","长虹","有方","诺亚信","金国威","邦华","糖果","波导","欧正","兴华宝","海尔","亿通",
                "先科","黑鲨"),$"term_brand").otherwise("小品牌"))
        .withColumn("flag",$"label".cast(IntegerType))
        .drop("label").withColumnRenamed("flag","label")

    val modelData=tagData.na.fill(0,tagData.columns.filter(_ != "label"))
    val modelCols=modelData.columns.filter(!dropCols.flatten.contains(_))
    val sampleData=modelData.select(modelCols.map(col):_*)

    sampleData
  }
}


