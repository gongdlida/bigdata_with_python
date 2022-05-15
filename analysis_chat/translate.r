read_file <- read.csv("analysis_chat/chatbot_dataset.csv",
header = T, sep = ",", as.is = T)
str(read_file)

length(table(read_file[, 3]))
read_file[read_file$user,]
chat_request <- read_file[read_file$user, ]
chat_request
chat_response <- read_file[, 1]

write.csv(read_file, file= "analysis_chat/chagned.csv",quote = F)
