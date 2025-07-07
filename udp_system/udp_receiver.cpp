#include <iostream>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>

#pragma pack(push, 1) // 保证字节对齐和发送端一致

struct _OWL_Marker {
    unsigned int ID;
    float x, y, z;
    float timestamp;
};

struct OWL_TrackerState {
    char name[32];
    int ID;
    float position[3];
    float pose[4];
    float headingY;
    _OWL_Marker markers[4];
};

#pragma pack(pop)

int main() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(7777);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sockfd, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("绑定失败");
        return 1;
    }

    std::cout << "📡 Listening on port 7777...\n";

    char buffer[1024];
    sockaddr_in sender;
    memset(&sender, 0, sizeof(sender));
    socklen_t senderLen = sizeof(sender);

    while (true) {
        ssize_t len = recvfrom(sockfd, buffer, sizeof(buffer), 0, (sockaddr*)&sender, &senderLen);
        if (len <= 0) continue;

        // 提取 tracker 数量
        int numTrackers;
        memcpy(&numTrackers, buffer, sizeof(int));
        if (numTrackers < 1) continue;

        // 提取 tracker 名称
        char trackerName[33] = {0};
        memcpy(trackerName, buffer + 4, 32);

        // 提取 tracker 数据
        OWL_TrackerState state;
        memcpy(&state, buffer + 4 + 32, sizeof(OWL_TrackerState));

        std::cout << "📍 Tracker: " << trackerName << "\n";
        std::cout << "  Position: (" << state.position[0] << ", " << state.position[1] << ", " << state.position[2] << ")\n";
        std::cout << " 4 bytes: (" << state.pose[0] << ", " << state.pose[1] << ", " << state.pose[2] << ", " << state.pose[3] << ")\n";
        std::cout << "  Heading Y: " << state.headingY << "\n";

        for (int i = 0; i < 4; ++i) {
            const _OWL_Marker& m = state.markers[i];
            std::cout << "  marker " << m.ID << ": " << m.x << ", " << m.y << ", " << m.z << "\n";
        }

        std::cout << "----------------------------\n";
    }
    close(sockfd);
    return 0;
}