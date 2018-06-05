
lazy val root = (project in file(".")).settings(
    name := "jariasPGM2018",
    organization := "es.jarias",
    version := "0.1.0",
    scalaVersion := "2.11.8",

    libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.3.0" % "provided",
    libraryDependencies += "org.apache.spark" % "spark-sql_2.11" % "2.3.0" % "provided",
    libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.3.0" % "provided"
)
.dependsOn(sparkBnc)
.aggregate(sparkBnc)

lazy val sparkBnc = RootProject(uri("https://github.com/jacintoarias/spark-bnc.git"))
