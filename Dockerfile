FROM mcr.microsoft.com/dotnet/runtime:7.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["DeerClassification.csproj", "./"]
RUN dotnet restore "DeerClassification.csproj"
COPY . .
WORKDIR "/src/"
RUN dotnet build "DeerClassification.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "DeerClassification.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
COPY /Assets /Assets
ENTRYPOINT ["dotnet", "DeerClassification.dll"]
