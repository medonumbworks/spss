import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
    application
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

application {
    mainClass.set("dev.cryptopulse.mcp.MainKt")
    applicationName = "cryptopulse-mcp"
    // slf4j-simple logs to stderr so stdout stays reserved for the MCP stdio transport.
    applicationDefaultJvmArgs = listOf("-Dorg.slf4j.simpleLogger.logFile=System.err", "-Dorg.slf4j.simpleLogger.defaultLogLevel=warn")
}

dependencies {
    implementation(project(":core"))
    implementation(libs.mcp.server)
    implementation(libs.ktor.server.cio)
    runtimeOnly(libs.slf4j.simple)

    testImplementation(kotlin("test"))
    testImplementation(libs.mcp.client)
    testImplementation(libs.ktor.client.cio)
    testImplementation(libs.kotlinx.coroutines.test)
}

tasks.test {
    useJUnitPlatform()
    systemProperty("org.slf4j.simpleLogger.logFile", "System.err")
    testLogging {
        events("failed", "skipped")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
