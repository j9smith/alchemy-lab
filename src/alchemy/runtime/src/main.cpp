# include <iostream>
# include <httplib.h>

int main(){
    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok\n", "text/plain");
    });

    std::cout << "listening on port 8000\n";
    svr.listen("0.0.0.0", 8000);
}