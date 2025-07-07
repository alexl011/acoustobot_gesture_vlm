// Windows UDP/keyboard/tracking demo
#include <winsock2.h>
#include <ws2tcpip.h>
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#pragma comment(lib, "ws2_32.lib")

// Minimal UDP sender/receiver
class UDPDriver {
    SOCKET sockfd;
    std::vector<std::string> clientIPs;
    int socketPort;
    struct sockaddr_in serverAddr;
public:
    UDPDriver() : sockfd(INVALID_SOCKET), socketPort(0) {}
    void initialise(const char* serverIP, int socketPort) {
        this->socketPort = socketPort;
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        memset(&serverAddr, 0, sizeof(serverAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(socketPort);
        inet_pton(AF_INET, serverIP, &serverAddr.sin_addr);
    }
    unsigned int addClient(const char* clientIP) {
        clientIPs.push_back(std::string(clientIP));
        return clientIPs.size() - 1;
    }
    void sendByte(unsigned char byte, int clientID = -1) {
        struct sockaddr_in destAddr;
        memset(&destAddr, 0, sizeof(destAddr));
        destAddr.sin_family = AF_INET;
        destAddr.sin_port = htons(socketPort);
        if (clientID >= 0 && clientID < (int)clientIPs.size()) {
            inet_pton(AF_INET, clientIPs[clientID].c_str(), &destAddr.sin_addr);
        } else {
            destAddr.sin_addr = serverAddr.sin_addr;
        }
        sendto(sockfd, (const char*)&byte, 1, 0, (struct sockaddr*)&destAddr, sizeof(destAddr));
    }
    void disconnect() {
        if (sockfd != INVALID_SOCKET) closesocket(sockfd);
        sockfd = INVALID_SOCKET;
    }
};

// Tracking data structure
struct TrackerState {
    float position[3];
    float pose[4];
    float headingY;
    struct _OWL_Marker {
        unsigned int ID;
        float x, y, z;
        float timestamp;
    } markers[4];
};

static const int MAX_SIZE_IN_BYTES = sizeof(int) + 12 * (32 * sizeof(char) + sizeof(TrackerState));
char recvBuff[MAX_SIZE_IN_BYTES];

#define PORT 7777
SOCKET sockPhaseSpace;
struct sockaddr_in recv_addr;

const int numMonaRobots = 1;
int Monas[numMonaRobots];

int main() {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
        printf("WSAStartup failed.\n");
        return 1;
    }

    // Setup UDP receive socket
    sockPhaseSpace = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sockPhaseSpace == INVALID_SOCKET) { perror("socket creation failed"); WSACleanup(); exit(1); }
    char broadcast = 'a';
    setsockopt(sockPhaseSpace, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));
    recv_addr.sin_family = AF_INET;
    recv_addr.sin_port = htons(PORT);
    recv_addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(sockPhaseSpace, (struct sockaddr*)&recv_addr, sizeof(recv_addr)) < 0) {
        perror("Error binding UDP socket.");
        closesocket(sockPhaseSpace);
        WSACleanup();
        exit(1);
    }

    // Setup UDP sender
    UDPDriver udp;
    int socketPort = 54007;
    char serverIP[] = "192.168.167.255";
    char Mona_addresses[numMonaRobots][14] = {
        "192.168.0.222"
    };
    udp.initialise(serverIP, socketPort);
    for (int i = 0; i < numMonaRobots; i++) {
        Monas[i] = udp.addClient(Mona_addresses[i]);
    }

    printf("Listening for tracking data on UDP port %d. Press keys to send commands. Press x to exit.\n", PORT);

    float robot_pos[3] = {NAN, NAN, NAN};
    float object_pos[3] = {NAN, NAN, NAN};
    float robot_heading = NAN;
    const char* ROBOT_TRACKER_NAME = "AcoustoBot1";
    const char* OBJECT_TRACKER_NAME = "User1";
    const float REACH_THRESHOLD = 0.05f;
    const float ANGLE_THRESHOLD = 10.0f;

    while (1) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(sockPhaseSpace, &readfds);
        struct timeval tv = {0, 10000}; // 10ms
        int rv = select((int)sockPhaseSpace+1, &readfds, NULL, NULL, &tv);
        if (rv > 0 && FD_ISSET(sockPhaseSpace, &readfds)) {
            sockaddr_in sender;
            int senderLen = sizeof(sender);
            int bytesReceived = recvfrom(sockPhaseSpace, recvBuff, MAX_SIZE_IN_BYTES, 0, (struct sockaddr*)&sender, &senderLen);
            if (bytesReceived > 0) {
                char* parsingIndex = recvBuff;
                int numTrackers = *((int*)parsingIndex);
                parsingIndex += sizeof(int);

                robot_pos[0] = robot_pos[1] = robot_pos[2] = NAN;
                object_pos[0] = object_pos[1] = object_pos[2] = NAN;
                robot_heading = NAN;

                for (int t = 0; t < numTrackers; t++) {
                    char trackerName[33] = {0};
                    memcpy(trackerName, parsingIndex, 32);
                    parsingIndex += 32;
                    TrackerState curState;
                    memcpy(&curState, parsingIndex, sizeof(TrackerState));
                    parsingIndex += sizeof(TrackerState);

                    printf("Tracker: %s | Pos: (%.3f, %.3f, %.3f) | Heading: %.2f\n",
                        trackerName, curState.position[0], curState.position[1], curState.position[2], curState.headingY);

                    if (strcmp(trackerName, ROBOT_TRACKER_NAME) == 0) {
                        memcpy(robot_pos, curState.position, sizeof(robot_pos));
                        robot_heading = curState.headingY;
                    }
                    if (strcmp(trackerName, OBJECT_TRACKER_NAME) == 0) {
                        memcpy(object_pos, curState.position, sizeof(object_pos));
                    }
                }

                if (!std::isnan(robot_pos[0]) && !std::isnan(object_pos[0]) && !std::isnan(robot_heading)) {
                    float dx = object_pos[0] - robot_pos[0];
                    float dy = object_pos[1] - robot_pos[1];
                    float dist = std::sqrt(dx * dx + dy * dy);
                    float desired_heading = atan2(dy, dx) * 180.0f / M_PI;

                    float heading_error = desired_heading - robot_heading;
                    while (heading_error > 180.0f) heading_error -= 360.0f;
                    while (heading_error < -180.0f) heading_error += 360.0f;

                    if (dist <= REACH_THRESHOLD) {
                        udp.sendByte('S', Monas[0]);
                        printf("Robot reached the object!\n");
                    } else if (std::abs(heading_error) > ANGLE_THRESHOLD) {
                        if (heading_error > 0)
                            udp.sendByte('L', Monas[0]);
                        else
                            udp.sendByte('R', Monas[0]);
                        printf("Turning robot (error %.2f°)...\n", heading_error);
                    } else {
                        udp.sendByte('F', Monas[0]);
                        printf("Moving forward. Distance: %.3f, Heading error: %.2f\n", dist, heading_error);
                    }
                }
            }
        }

        if (_kbhit()) {
            int c = _getch();
            if (c == 'x' || c == 'X') break;
            for (int i = 0; i < numMonaRobots; i++) {
                udp.sendByte((unsigned char)c, Monas[i]);
            }
            printf("Sent command '%c' to all robots.\n", c);
        }
    }

    closesocket(sockPhaseSpace);
    udp.disconnect();
    WSACleanup();
    printf("Exited.\n");
    return 0;
}
