import express from "express";
import dotenv from "dotenv"
dotenv.config()
import authRouter from "./routes/auth.js"
import connectDB from "./config/mongoDB.js"
const app = express();


app.use(express.json())

app.use("/api/v1/auth", authRouter)

app.get("/", (req, res) => {
    res.send("Hello World!");
});

app.listen(5000, () => {
    console.log("Server is running on port 5000");
    connectDB();
});