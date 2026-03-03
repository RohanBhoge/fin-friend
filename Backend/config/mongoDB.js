import mongoose from "mongoose";


const connectDB = async () => {
    mongoose.connection.on("connected", () => {
        console.log("Connected to MongoDB");
    })
    try {
        await mongoose.connect(process.env.MONGO_URI)
    }
    catch (error) {
        console.log("Error connecting to MongoDB", error);
        process.exit(1);
    }
}

export default connectDB;
