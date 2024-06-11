// The code in this file will load on every page of your site
import { getResponseGCP } from 'backend/google-cloud';
import { sendChatMessage } from 'backend/wix-chat';
// The code in this file will load on every page of your site

$w.onReady(function () {
    $w('#wixChatAgent').onMessageSent(async (message) => {
        const messageText = message.payload.text;
        const channelId = message.channelId;
        // logMessage("user", messageText);
		const query = {"id":channelId, "request":messageText}
        // console.log(messages);
        const answer = await getResponseGCP(query);
        // logMessage(answer.role, answer.content);
        console.log(answer);
        console.log(channelId);
        sendChatMessage(answer, channelId);
    })
});
