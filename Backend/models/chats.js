import mongoose from "mongoose"

const chatSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User"
    },
    messages: [
        {
            role: {
                type: String,
                required: true
            },
            content: {
                type: String,
                required: true
            }
        }
    ]
}, { timestamps: true });

const chatModel = mongoose.model("Chat", chatSchema);
export default chatModel;